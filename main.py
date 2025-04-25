import argparse
import multiprocessing
from utils.processor import ArxivProcessor

# Set start method for multiprocessing
multiprocessing.set_start_method('spawn', force=True)  # CUDA requires spawn

def main():
    parser = argparse.ArgumentParser(description="arXiv PDF to Markdown Converter (Batch Worker)")
    parser.add_argument("--month", required=True, help="Month (01-12)")
    parser.add_argument("--year", required=True, help="Year (e.g., 21 for 2021)")
    parser.add_argument("--output", default="./data", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of papers processed per worker process")
    parser.add_argument("--prefetch", type=int, default=3, help="Download queue size factor (prefetch * batch_size)")
    parser.add_argument("--timeout-per-paper", type=int, default=300,
                        help="Estimated timeout PER PAPER in seconds. Total batch timeout will be this * batch_size.")
    args = parser.parse_args()

    processor = ArxivProcessor(
        month=args.month,
        year=args.year,
        output_dir=args.output,
        batch_size=args.batch_size,
        prefetch_factor=args.prefetch,
        timeout=args.timeout_per_paper
    )

    processor.run()

if __name__ == "__main__":
    main()
