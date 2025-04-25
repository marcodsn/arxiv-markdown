# arxiv-markdown

A repository for converting arXiv papers to markdown format and creating a machine-learning ready dataset.

## Description

This repository contains code for extracting and converting open-access papers from [arXiv](https://arxiv.org) to markdown format using [docling](https://github.com/docling-project/docling). The extracted markdown files include embedded LaTeX formulas, code blocks, and references to images hosted on Cloudflare R2.

The primary goal is to create a large-scale, high-quality dataset of academic papers in markdown format that can be used for various machine learning applications, including the expansion of the [academic-chains](https://huggingface.co/datasets/marcodsn/academic-chains) dataset.

## Dataset

The resulting dataset is available on HuggingFace:
- **Dataset:** [marcodsn/arxiv-markdown](https://huggingface.co/datasets/marcodsn/arxiv-markdown)

Currently, the dataset contains initial entries from August 2024, with continuous updates as more papers are processed.

## Repository Structure

```
arxiv-markdown/
├── main.py                   # Entry point script with argument parsing, run this to start generating markdown files
│
├── utils/                    # Utility modules
│   ├── __init__.py           # Makes utils a proper package
│   ├── storage.py            # Cloud storage functions for R2
│   ├── conversion.py         # Document conversion utilities
│   └── processor.py          # Main processor class
│
├── data/                     # Data directory
│   ├── checkpoints/          # Processing checkpoints
│   └── jsonls/               # Processed JSON line files
│
├── scripts/                  # Additional utility scripts
│   └── create_dataset.py     # Dataset creation and upload script
│
└── README.md                 # This file
```

## Conversion Pipeline

The conversion pipeline uses the following docling configuration:

```python
pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE  # 2.0
pipeline_options.generate_page_images = True  # Necessary for the extraction of figures (as far as I understand)
pipeline_options.generate_picture_images = True  # For the extraction of figures
pipeline_options.do_code_enrichment = True  # For obtaining code blocks
pipeline_options.do_formula_enrichment = True  # For converting formulas to LaTeX
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
```

### Image Storage

As of April 24, 2025, images are no longer embedded as base64 in the markdown files. Instead, they are uploaded to a Cloudflare R2 instance and referenced in the markdown using standard markdown image syntax: `![Image](https://example.com/image.png)`.

## Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/marcodsn/arxiv-markdown.git
cd arxiv-markdown

# Install dependencies
pip install -r requirements.txt
```

### Running the Extraction

```bash
# Extract papers from a specific month/year
python main.py --month 8 --year 24
```

### Processing Options

- `--month`: Month to process (1-12)
- `--year`: Year to process (e.g., 24 for 2024)
- `--output`: Output directory for markdown files
- `--batch-size`: Number of papers to process in each batch (default: 4)
- `--prefetch`: Prefetch factor for downloader thread (default: 3)
- `--timeout-per-paper`: Timeout in seconds per paper processing (default: 240)

## Known Limitations

- **Conversion Speed**: The extraction process with formula and code enrichment plus image extraction is relatively slow, even on modern GPUs (currently using an RTX 3090).
- **Timeout Issues**: Due to a [known bug in docling](https://github.com/docling-project/docling/issues/1283), a timeout of 240 seconds per paper is used, which may cause some papers to be skipped.
- **Extraction Quality**: While docling provides high-quality conversion, some extraction glitches may occur, especially in complex tables.

## Future Work

- Implementing support for batched inference in docling
- Exploring VLM-based approaches for extraction
- Expanding the dataset to cover more arXiv categories and years
- Improving the extraction pipeline performance

## Acknowledgements

A big thank you to:
- arXiv and to all the authors of the open-access papers
- The docling project for their document conversion tools
- Contributors and supporters of the academic-chains dataset

## Licensing

- **Repository Code**: This code is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **Dataset**: The dataset itself is licensed under the [CC-BY-4.0 License](https://creativecommons.org/licenses/by/4.0/)

## Citation

```bibtex
@misc{marcodsn_2025_arxivmarkdown,
    title = {arxiv-markdown},
    author = {Marco De Santis},
    month = {April},
    year = {2025},
    url = {https://huggingface.co/datasets/marcodsn/arxiv-markdown}
}
```

## Contributing

Optimizations and general suggestions are very welcome! This repository is under active development.

We are also looking for people willing to mirror our R2 instance for added redundancy.
