import boto3
import os
from dotenv import load_dotenv

load_dotenv()

def upload_to_r2(image_data, filename, content_type='image/jpeg'):
    """Upload image to Cloudflare R2 and return the public URL"""
    # Initialize R2 client (ideally once per batch outside this function)
    s3_client = boto3.client(
        's3',
        endpoint_url = os.getenv("R2_ENDPOINT"),
        aws_access_key_id = os.getenv("R2_KEY_ID"),
        aws_secret_access_key = os.getenv("R2_KEY_SECRET"),
        region_name = os.getenv("R2_REGION", "auto")
    )

    # Upload the image
    s3_client.put_object(
        Bucket = 'arxiv-markdown-images',
        Key = f"{filename}",
        Body = image_data,
        ContentType = content_type
    )

    # Return the public URL
    return f"https://ac.marcodsn.me/arxiv-markdown-images/{filename}"
