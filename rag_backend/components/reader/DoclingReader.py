from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.document import InputDocument
from io import BytesIO
import base64
import os
from typing import List, Optional

from rag_backend.components.interfaces import Reader
from rag_backend.components.document import Document
from rag_backend.server.types import FileConfig


class DoclingReader(Reader):
    """Reader that uses Docling to convert various document formats to text"""
    
    def __init__(self):
        self.name = "Docling"
        self.description = "Uses Docling to convert various document formats to text"
        self.library = ["docling"]
        self.available = True
        self.variables = []
        self.config = {
            "artifacts_path": {
                "description": "Path to local Docling models",
                "values": ["DOCLING_ARTIFACTS_PATH", "custom_path"],
                "value": "DOCLING_ARTIFACTS_PATH"
            },
            "enable_remote_services": {
                "description": "Enable remote services for processing",
                "values": [True, False],
                "value": False
            },
            "do_table_structure": {
                "description": "Enable table structure recognition",
                "values": [True, False],
                "value": True
            },
            "table_structure_mode": {
                "description": "TableFormer mode for table structure recognition",
                "values": ["FAST", "ACCURATE"],
                "value": "ACCURATE"
            },
            "max_num_pages": {
                "description": "Maximum number of pages to process",
                "values": ["unlimited", "custom"],
                "value": "unlimited"
            },
            "max_file_size": {
                "description": "Maximum file size in bytes",
                "values": ["unlimited", "custom"],
                "value": "unlimited"
            },
            "use_models": {
                "description": "Whether to use Docling models for processing",
                "values": [True, False],
                "value": True
            }
        }
        self.requires_library = ["docling"]
        self.requires_env = []
        
    def _get_artifacts_path(self) -> Optional[str]:
        """Get artifacts path from environment or config"""
        return os.getenv("DOCLING_ARTIFACTS_PATH")
        
    def _get_config_value(self, config, key: str, default: str = "true") -> str:
        """Get a configuration value from either a dict or Pydantic model"""
        try:
            if isinstance(config, dict):
                # Handle dictionary config
                if key in config:
                    if isinstance(config[key], dict) and "value" in config[key]:
                        return str(config[key]["value"])
                    return str(config[key])
                return default
            else:
                # Handle Pydantic model
                if hasattr(config, key):
                    setting = getattr(config, key)
                    if hasattr(setting, "value"):
                        return str(setting.value)
                    return str(setting)
                return default
        except Exception:
            return default
        
    def _create_converter(self, config) -> DocumentConverter:
        """Create a DocumentConverter with the given configuration"""
        use_models = self._get_config_value(config, "use_models", "true").lower() == "true"
        enable_remote = self._get_config_value(config, "enable_remote_services", "false").lower() == "true"
        do_table_structure = self._get_config_value(config, "do_table_structure", "true").lower() == "true"
        table_mode = self._get_config_value(config, "table_structure_mode", "ACCURATE")
        
        if use_models:
            artifacts_path = self._get_artifacts_path()
            if not artifacts_path:
                raise Exception("DOCLING_ARTIFACTS_PATH environment variable not set when using models")
                
            pipeline_options = PdfPipelineOptions(
                artifacts_path=artifacts_path,
                enable_remote_services=enable_remote,
                do_table_structure=do_table_structure
            )
            
            if do_table_structure:
                pipeline_options.table_structure_options.mode = (
                    TableFormerMode.FAST if table_mode == "FAST" else TableFormerMode.ACCURATE
                )
                
            return DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
        else:
            return DocumentConverter()

    async def load(self, config, fileConfig: FileConfig) -> List[Document]:
        """
        Load and convert a document using Docling
        
        Args:
            config: Configuration for the reader
            fileConfig (FileConfig): Configuration for the file to be loaded
            
        Returns:
            List[Document]: List of converted documents
        """
        try:
            max_pages = self._get_config_value(config, "max_num_pages", "unlimited")
            max_size = self._get_config_value(config, "max_file_size", "unlimited")
            
            if max_pages != "unlimited":
                max_pages = int(max_pages)
            if max_size != "unlimited":
                max_size = int(max_size)
                
            converter = self._create_converter(config)
            
            if fileConfig.isURL:
                source = fileConfig.filename
            else:
                content = base64.b64decode(fileConfig.content)
                source = DocumentStream(
                    name=fileConfig.filename,
                    stream=BytesIO(content)
                )
                
            result = converter.convert(
                source,
                max_num_pages=1000 if max_pages == "unlimited" else max_pages,
                max_file_size=1000000000 if max_size == "unlimited" else max_size
            )
            
            markdown_content = result.document.export_to_markdown()
            
            doc = Document(
                title=fileConfig.filename,
                content=markdown_content,
                extension=fileConfig.extension,
                fileSize=len(markdown_content.encode()),
                labels=fileConfig.labels,
                source=fileConfig.source,
                metadata=fileConfig.metadata,
                meta={}
            )
            
            return [doc]
            
        except Exception as e:
            raise Exception(f"Docling conversion failed: {str(e)}")
            
    def get_meta(self, environment_variables: dict, libraries: dict) -> dict:
        """
        Get metadata about the reader
        
        Args:
            environment_variables (dict): Available environment variables
            libraries (dict): Available libraries
            
        Returns:
            dict: Metadata about the reader
        """
        return {
            "name": self.name,
            "type": "Reader",
            "description": self.description,
            "library": self.library,
            "available": self.available and all(libraries.get(lib, False) for lib in self.requires_library),
            "variables": self.variables,
            "config": self.config
        }