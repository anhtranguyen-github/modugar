import asyncio
import base64
import os
import re
import warnings
import aiohttp
from pathlib import Path
from rag_backend.components.reader.DoclingReader import DoclingReader
from rag_backend.server.types import FileConfig

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def sanitize_filename(filename):
    """Convert URL or path to a valid filename"""
    # Extract filename from URL or path
    basename = os.path.basename(filename)
    # Remove invalid characters
    safe_name = re.sub(r"[^\w\-_.]", "_", basename)
    # Remove any double extensions
    name, ext = os.path.splitext(safe_name)
    if ext.lower() == ".md":
        return name
    return safe_name


async def fetch_url_content(url):
    """Fetch content from URL with error handling"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    raise Exception(f"HTTP {response.status}: {response.reason}")
    except Exception as e:
        raise Exception(f"Failed to fetch URL {url}: {str(e)}")


def read_local_file(filepath):
    """Read content from local file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # If UTF-8 fails, try reading as binary and encode as base64
        with open(filepath, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        raise Exception(f"Failed to read file {filepath}: {str(e)}")


async def process_file(reader, filename, content="", is_url=False):
    """Process a single file and save output as markdown"""
    try:
        print(f"\nProcessing file: {filename}")
        print(f"File type: {'URL' if is_url else 'Local file'}")
        
        # Get content based on file type
        if is_url:
            try:
                print("Fetching URL content...")
                content = await fetch_url_content(filename)
                print(f"Successfully fetched {len(content)} bytes")
            except Exception as e:
                print(f"Error fetching URL {filename}: {str(e)}")
                return
        else:
            try:
                print("Reading local file...")
                if not os.path.exists(filename):
                    print(f"File not found: {filename}")
                    return
                content = read_local_file(filename)
                print(f"Successfully read {len(content)} bytes")
            except Exception as e:
                print(f"Error reading file {filename}: {str(e)}")
                return

        # Create file config
        print("Creating file config...")
        file_config = FileConfig(
            filename=filename,
            content=content,
            extension=os.path.splitext(filename)[1],
            file_size=len(content.encode()) if isinstance(content, str) else len(content),
            labels=["test"],
            source="url" if is_url else "local",
            metadata=f"Test file: {filename}",
            fileID=f"test_{filename}",
            isURL=is_url,
            overwrite=False,
            rag_config={
                "Reader": {
                    "selected": "Docling",
                    "components": {
                        "Docling": {
                            "name": "Docling",
                            "type": "Reader",
                            "description": "Test reader",
                            "library": ["docling"],
                            "available": True,
                            "variables": [],
                            "config": {
                                "use_models": {
                                    "type": "boolean",
                                    "description": "Whether to use Docling models for processing",
                                    "values": ["true", "false"],
                                    "value": "false"
                                }
                            }
                        }
                    }
                }
            },
            status="READY",
            status_report={}
        )

        # Pass the config directly
        print("Loading document with DoclingReader...")
        config = {
            "use_models": {
                "type": "boolean",
                "description": "Whether to use Docling models for processing",
                "values": ["true", "false"],
                "value": "false"
            }
        }
        documents = await reader.load(config, file_config)
        
        # Save output
        if documents:
            safe_name = sanitize_filename(filename)
            if not safe_name:  # Handle empty filenames
                safe_name = "untitled"
            output_filename = f"output_{safe_name}.md"
            print(f"Saving output to {output_filename}...")
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(documents[0].content)
            print(f"Successfully processed {filename} -> {output_filename}")
            print(f"Output size: {len(documents[0].content)} bytes")
        else:
            print(f"No content extracted from {filename}")

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        import traceback
        print("Full error traceback:")
        print(traceback.format_exc())


async def main():
    reader = DoclingReader()
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Test files - using specific files
    files = [
        {
            "filename": "https://github.com/microsoft/tinytroupe",
            "is_url": True
        },
        {
            "filename": "/mnt/c/Users/tra01/OneDrive/Desktop/rag/myrag/data/Bài tập 1.pdf",
            "is_url": False
        },
        {
            "filename": "data/Convert_chuỗi_ảnh_360_sang_không_gian_3D_points_cloud.docx",
            "is_url": False
        }
    ]

    # Process each file
    for file in files:
        print(f"\nProcessing {file['filename']}...")
        await process_file(reader, file["filename"], is_url=file["is_url"])


if __name__ == "__main__":
    print("Starting DoclingReader file processing tests...")
    asyncio.run(main())
    print("\nAll tests completed!")
