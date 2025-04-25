import os
import json
import queue
import tempfile
import threading
import subprocess
from multiprocessing import Process, Queue
import time
import shutil
from pathlib import Path
from utils.conversion import batch_convert_worker

class ArxivProcessor:
    def __init__(self, month, year, output_dir, batch_size=8, prefetch_factor=3, timeout=300):
        self.month = month.zfill(2)
        self.year = year
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        # IMPORTANT: This timeout now applies to the WHOLE BATCH
        self.batch_timeout = timeout * batch_size # Start with a scaled timeout, adjust as needed
        print(f"Setting BATCH timeout to {self.batch_timeout} seconds ({timeout}s per paper * {batch_size} papers)")

        self.temp_base_dir = tempfile.mkdtemp(prefix="arxiv_processing_")
        print(f"Using temporary directory: {self.temp_base_dir}")
        self.dataset_path = self.output_dir / "jsonls" / f"arxiv_{year}{self.month}.jsonl"
        self.checkpoint_path = self.output_dir / "checkpoints" / f"arxiv_{year}{self.month}.checkpoint"
        self.paper_queue = queue.Queue(maxsize=prefetch_factor * batch_size)
        self.processed_ids = set()
        self.load_checkpoint()
        self.worker_process_counter = 0 # To give workers IDs for logging

    def __del__(self):
        try:
            if hasattr(self, 'temp_base_dir') and os.path.exists(self.temp_base_dir):
                shutil.rmtree(self.temp_base_dir, ignore_errors=True)
                print(f"Cleaned up base temporary directory: {self.temp_base_dir}")
        except Exception as e:
             print(f"Error cleaning up base temp dir: {e}")

    def load_checkpoint(self):
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'r') as f:
                self.processed_ids = set(line.strip() for line in f)
            print(f"Resuming from checkpoint with {len(self.processed_ids)} processed papers")

    def update_checkpoint(self, paper_id):
        with open(self.checkpoint_path, 'a') as f:
            f.write(f"{paper_id}\n")

    def list_papers(self):
        cmd = f"gsutil ls gs://arxiv-dataset/arxiv/arxiv/pdf/{self.year}{self.month}/*.pdf"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error listing papers: {result.stderr}")
            return []

        paper_urls = result.stdout.strip().split('\n')
        paper_urls = [url for url in paper_urls if url.strip()]  # Filter out empty strings

        # Group papers by base ID (without version suffix)
        paper_versions = {}
        for url in paper_urls:
            if not url:
                continue

            try:
                filename = url.split('/')[-1]
                paper_id = filename.replace('.pdf', '')

                # Extract base ID and version
                if 'v' in paper_id:
                    base_id, version_str = paper_id.rsplit('v', 1)
                    try:
                        version_num = int(version_str)
                    except ValueError:
                        print(f"Warning: Invalid version in {paper_id}, treating as v1")
                        version_num = 1
                else:
                    base_id = paper_id
                    version_num = 1

                # Keep track of highest version for each base ID
                if base_id not in paper_versions or version_num > paper_versions[base_id]['version']:
                    paper_versions[base_id] = {
                        'version': version_num,
                        'full_id': paper_id,
                        'url': url
                    }
            except Exception as e:
                print(f"Warning: Could not parse paper ID from URL: {url} - {e}")

        # Select only the latest version of each paper that hasn't been processed
        new_papers = []
        for base_id, paper_info in paper_versions.items():
            full_id = paper_info['full_id']
            if full_id not in self.processed_ids:
                new_papers.append({"arxiv_id": full_id, "url": paper_info['url']})

        print(f"Found {len(paper_urls)} total PDF files")
        print(f"Identified {len(paper_versions)} unique papers (after grouping versions)")
        print(f"Found {len(new_papers)} new papers to process (latest versions only)")

        return new_papers

    def download_paper(self, paper):
        paper_dir = None
        try:
            paper_dir = tempfile.mkdtemp(prefix=f"{paper['arxiv_id']}_", dir=self.temp_base_dir)
            pdf_path = Path(paper_dir) / f"{paper['arxiv_id']}.pdf"
            # print(f"Downloading {paper['arxiv_id']} to {pdf_path}") # Less verbose download
            cmd = f"gsutil cp {paper['url']} {pdf_path}"
            result = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                 raise Exception(f"gsutil failed: {result.stderr}")
            # print(f"Successfully downloaded {paper['arxiv_id']}") # Less verbose download
            return {**paper, "local_path": str(pdf_path), "temp_dir": paper_dir}
        except subprocess.TimeoutExpired:
            print(f"Timeout downloading {paper['arxiv_id']}")
            if paper_dir and os.path.exists(paper_dir): shutil.rmtree(paper_dir, ignore_errors=True)
            return None
        except Exception as e:
            print(f"Error downloading {paper['arxiv_id']}: {e}")
            if paper_dir and os.path.exists(paper_dir): shutil.rmtree(paper_dir, ignore_errors=True)
            return None

    def downloader_thread(self, papers_metadata):
        download_count = 0
        total_papers = len(papers_metadata)
        queue_max_size = self.paper_queue.maxsize
        print(f"[Downloader] Started. Target: {total_papers} papers. Prefetch queue max size: {queue_max_size}")
        try:
            for paper in papers_metadata:
                try:
                    paper_data = self.download_paper(paper)
                    if paper_data is not None:
                        if self.paper_queue.full():
                            current_size = self.paper_queue.qsize() # Get current size (should be maxsize)
                            print(f"[Downloader] Prefetch queue is FULL (Size: {current_size}/{queue_max_size}). Waiting for processor to consume items...")

                        self.paper_queue.put(paper_data)
                        download_count += 1
                    else:
                         print(f"Marking failed download {paper.get('arxiv_id', 'unknown')} as processed.")
                         self.update_checkpoint(paper.get('arxiv_id', 'unknown'))
                except Exception as e:
                    print(f"Error in download loop for {paper.get('arxiv_id', 'unknown')}: {e}")
                    self.update_checkpoint(paper.get('arxiv_id', 'unknown'))
        finally:
            print(f"Downloader thread finished. Downloaded {download_count} papers.")
            self.paper_queue.put(None)

    def _handle_batch_failure(self, paper_batch_info, error_message):
        """Handle failed batch processing by cleaning up and updating checkpoints"""
        error_result_list = []
        for paper_info in paper_batch_info:
            arxiv_id = paper_info.get("arxiv_id", "unknown")
            error_result_list.append({"arxiv_id": arxiv_id, "error": error_message})
            self.update_checkpoint(arxiv_id)  # Mark as processed to avoid retrying

            # Clean up temp directory if it exists
            if "temp_dir" in paper_info and os.path.exists(paper_info["temp_dir"]):
                shutil.rmtree(paper_info["temp_dir"], ignore_errors=True)

        return error_result_list

    def _terminate_process(self, proc, worker_id):
        """Safely terminate a process with escalation if needed"""
        print(f"Terminating batch process {proc.pid} (ID: {worker_id})")
        proc.terminate()
        proc.join(5)

        if proc.is_alive():
            print(f"Force killing batch process {proc.pid} (SIGKILL)")
            try:
                os.kill(proc.pid, 9)
                proc.join(1)
            except:
                pass

    def convert_batch_with_process_timeout(self, paper_batch_info, **kwargs):
        """
        Runs the BATCH conversion in a single separate process with timeout handling.
        """
        # Use the class's batch timeout if not overridden
        timeout = kwargs.get('timeout', self.batch_timeout)
        result_queue = Queue()
        self.worker_process_counter += 1
        worker_id = self.worker_process_counter

        # Create process for batch worker
        proc = Process(
            target=batch_convert_worker,
            args=(paper_batch_info, result_queue, worker_id)
        )

        start_time = time.time()
        proc.start()
        print(f"Started BATCH worker process {proc.pid} (ID: {worker_id}) for {len(paper_batch_info)} papers.")

        try:
            # Wait for the process to finish or the BATCH timeout
            batch_results = result_queue.get(timeout=timeout)
            proc.join()  # Ensure process is joined after getting result
            elapsed = time.time() - start_time
            print(f"Batch worker process {proc.pid} (ID: {worker_id}) finished in {elapsed:.2f}s")
            return batch_results  # Return the list of results from the queue

        except queue.Empty:
            print(f"BATCH CONVERSION TIMEOUT after {timeout}s for worker {proc.pid} (ID: {worker_id})")
            self._terminate_process(proc, worker_id)
            return self._handle_batch_failure(
                paper_batch_info,
                f"Batch timeout after {timeout}s"
            )

        except Exception as e:
            print(f"Error managing BATCH process {proc.pid} (ID: {worker_id}): {e}")
            self._terminate_process(proc, worker_id)
            return self._handle_batch_failure(
                paper_batch_info,
                f"Batch process management error: {str(e)}"
            )

    def _initialize_dataset(self):
        """Initialize the dataset file if it doesn't exist"""
        if not self.dataset_path.exists():
            with open(self.dataset_path, "w") as _:
                pass

    def _gather_batch(self, downloader):
        """Gather a batch of papers from the queue"""
        current_batch_info = []
        batch_full = False
        done_downloading = False
        paper_data = None

        try:
            # Gather a full batch
            while len(current_batch_info) < self.batch_size:
                paper_data = self.paper_queue.get(timeout=600)  # Wait for papers
                if paper_data is None:  # Sentinel found
                    done_downloading = True  # Signal that downloading is complete
                    break  # Stop filling batch
                current_batch_info.append(paper_data)
            else:  # Executed if the while loop finished without break (i.e. batch is full)
                batch_full = True

        except queue.Empty:
            print("Warning: Timed out waiting for paper from download queue.")
            if not downloader.is_alive() and self.paper_queue.empty():
                print("Downloader finished and queue is empty.")
                done_downloading = True

        return current_batch_info, batch_full, done_downloading, paper_data

    def _process_batch_results(self, batch_results, current_batch_info):
        """Process the results from a batch conversion"""
        processed_in_batch_count = 0

        # Process results for each paper in the batch
        if not isinstance(batch_results, list):
            print(f"Error: Unexpected result type from batch worker: {type(batch_results)}")
            # Mark all papers in the batch as processed to avoid retries
            for paper_info in current_batch_info:
                self.update_checkpoint(paper_info.get("arxiv_id", "unknown_batch_error"))
            return 0

        for result in batch_results:
            arxiv_id = result.get("arxiv_id", "unknown_result")

            if "error" in result:
                print(f"Failed to process {arxiv_id}: {result['error']}")
            elif "markdown" in result:
                # Append successful conversion to JSONL
                with open(self.dataset_path, "a") as output_file:
                    output_file.write(json.dumps(result) + "\n")
                processed_in_batch_count += 1
            elif "batch_error" in result:
                print(f"Batch worker failed critically: {result['batch_error']}")

            # Update checkpoint if not a complete batch failure
            if "batch_error" not in result:
                self.update_checkpoint(arxiv_id)

        return processed_in_batch_count

    def _print_summary(self, total_estimate, papers_attempted, papers_successful):
        """Print processing summary"""
        print("\n--- Processing Summary ---")
        print(f"Month/Year: {self.year}-{self.month}")
        print(f"Initial estimate of papers to process: {total_estimate}")
        print(f"Total papers attempted (including previous runs): {papers_attempted}")
        print(f"Papers successfully converted in this run: {papers_successful}")
        print(f"Results saved to: {self.dataset_path}")
        print(f"Checkpoint file: {self.checkpoint_path}")

    def run(self):
        """Main execution flow using batch workers"""
        self._initialize_dataset()

        papers_metadata = self.list_papers()
        if not papers_metadata:
            print("No new papers to process based on checkpoint.")
            self.__del__()
            return

        # Start downloader thread
        downloader = threading.Thread(target=self.downloader_thread, args=(papers_metadata,))
        downloader.daemon = True
        downloader.start()

        total_to_process_estimate = len(papers_metadata)
        papers_processed_successfully = 0
        papers_attempted = len(self.processed_ids)

        processing_complete = False
        while not processing_complete:
            # Gather a batch of papers to process
            current_batch_info, batch_full, done_downloading, paper_data = self._gather_batch(downloader)

            # Process the gathered batch if any papers were collected
            if current_batch_info:
                print(f"\nProcessing batch of {len(current_batch_info)} papers...")
                start_time = time.time()

                # Process the whole batch in one worker process
                batch_results = self.convert_batch_with_process_timeout(
                    current_batch_info,
                    timeout=self.batch_timeout
                )

                # Process and record results
                processed_count = self._process_batch_results(batch_results, current_batch_info)
                papers_processed_successfully += processed_count
                papers_attempted += len(current_batch_info)

                elapsed = time.time() - start_time
                print(f"Batch finished in {elapsed:.2f}s. "
                      f"Attempted: {papers_attempted}. "
                      f"Successful this run: {papers_processed_successfully}.")

            # Check if processing is complete
            if not batch_full and done_downloading:
                processing_complete = True

        # Wait for downloader thread to complete
        print("Waiting for downloader thread to complete...")
        downloader.join(timeout=10)
        if downloader.is_alive():
            print("Downloader thread still active.")

        # Print final summary
        self._print_summary(total_to_process_estimate, papers_attempted, papers_processed_successfully)
        self.__del__()
