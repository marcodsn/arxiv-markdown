import os
import io
import uuid
import time
import shutil
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import ImageRefMode, PictureItem
from storage import upload_to_r2

IMAGE_RESOLUTION_SCALE = 2.0

def batch_convert_worker(paper_batch_info, result_queue, worker_id):
    """
    Worker function to be run in a separate process.
    Initializes ONE DocumentConverter and processes a BATCH of papers.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Only using GPU 0

    results_list = []
    converter = None # Initialize later inside try block

    print(f"[Worker {worker_id}] Started. Processing batch of {len(paper_batch_info)} papers.")
    start_time_batch = time.time()

    try:
        # Initialize converter ONCE for the batch
        print(f"[Worker {worker_id}] Initializing DocumentConverter...")
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        pipeline_options.do_code_enrichment = True
        pipeline_options.do_formula_enrichment = True
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        print(f"[Worker {worker_id}] DocumentConverter initialized.")

        # Loop through papers in the batch
        for i, paper_info in enumerate(paper_batch_info):
            arxiv_id = paper_info.get('arxiv_id', 'unknown')
            local_path = paper_info.get('local_path')
            temp_dir = paper_info.get('temp_dir') # Get temp dir for cleanup
            print(f"[Worker {worker_id}] Processing paper {i+1}/{len(paper_batch_info)}: {arxiv_id}")
            start_time_paper = time.time()
            paper_result = None

            if not local_path or not os.path.exists(local_path):
                 print(f"[Worker {worker_id}] Error: PDF path missing or not found for {arxiv_id} at {local_path}")
                 paper_result = {"arxiv_id": arxiv_id, "error": "PDF path missing or invalid."}
                 results_list.append(paper_result)
                 # Still attempt cleanup of temp_dir if it exists
                 if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                 continue # Skip to next paper

            try:
                # Convert using docling
                result = converter.convert(local_path)

                # Extract and upload images to Cloudflare R2
                image_urls = []

                # Process figures
                for element, _level in result.document.iterate_items():
                    if isinstance(element, PictureItem) and hasattr(element, 'get_image'):
                        # Generate unique ID for the figure
                        figure_id = str(uuid.uuid4())
                        image_filename = f"{arxiv_id}-figure-{figure_id}.jpg"

                        # Get image, convert to JPEG and upload
                        pil_img = element.get_image(result.document)
                        if not pil_img:
                            continue
                        jpeg_data = io.BytesIO()
                        pil_img.convert('RGB').save(jpeg_data, format='JPEG', quality=95)
                        jpeg_data.seek(0)

                        # Upload and store URL
                        r2_url = upload_to_r2(jpeg_data, image_filename, content_type='image/jpeg')
                        print(f"Uploaded image {image_filename} to R2, URL: {r2_url}")
                        image_urls.append(r2_url)

                markdown = result.document.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER, image_placeholder="<!-- image -->")
                # Replace <!-- image --> with the actual image URLs
                for i, url in enumerate(image_urls):
                    markdown = markdown.replace("<!-- image -->", f"![image]({url})", 1)

                # Create result object for this paper
                paper_result = {
                    "arxiv_id": arxiv_id,
                    "markdown": markdown,
                }
                results_list.append(paper_result)
                print(f"[Worker {worker_id}] Successfully converted {arxiv_id} in {time.time() - start_time_paper:.2f}s")

            except Exception as e:
                print(f"[Worker {worker_id}] Error converting {arxiv_id}: {e}")
                # Add error info for this specific paper
                results_list.append({"arxiv_id": arxiv_id, "error": str(e)})
            finally:
                # Clean up temporary directory for THIS paper
                if temp_dir and os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        # print(f"[Worker {worker_id}] Cleaned up temp dir: {temp_dir}")
                    except Exception as cleanup_e:
                        print(f"[Worker {worker_id}] Error cleaning up temp dir {temp_dir} for {arxiv_id}: {cleanup_e}")

        # Put the list of all results (successes and errors) for the batch onto the queue
        result_queue.put(results_list)
        total_batch_time = time.time() - start_time_batch
        print(f"[Worker {worker_id}] Finished batch of {len(paper_batch_info)} in {total_batch_time:.2f}s.")

    except Exception as batch_e:
        # Handle errors during converter initialization or other batch-level issues
        print(f"[Worker {worker_id}] CRITICAL BATCH ERROR: {batch_e}")
        # Try to send back whatever results were gathered, plus an error marker
        results_list.append({"batch_error": str(batch_e)})
        result_queue.put(results_list)
        # Clean up any remaining temp dirs for this batch if possible (best effort)
        for paper_info in paper_batch_info:
            if paper_info.get("temp_dir") and os.path.exists(paper_info.get("temp_dir")):
                 shutil.rmtree(paper_info.get("temp_dir"), ignore_errors=True)

    finally:
        # Optional: Explicitly release converter resources if needed
        # del converter
        pass
