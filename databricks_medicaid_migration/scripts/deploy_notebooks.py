"""
Deploy notebooks to Databricks workspace.

This script uploads all notebooks from the local notebooks/ directory
to the Databricks workspace.
"""

import os
import sys
from databricks_cli.sdk import ApiClient, WorkspaceService
from databricks_cli.workspace.api import WorkspaceApi
from pathlib import Path
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NotebookDeployer:
    """Deploy notebooks to Databricks workspace."""

    def __init__(self, host, token, workspace_path="/Medicaid/Notebooks"):
        """
        Initialize deployer.

        Args:
            host: Databricks workspace URL
            token: Personal access token
            workspace_path: Target path in workspace
        """
        self.api_client = ApiClient(host=host, token=token)
        self.workspace_api = WorkspaceApi(self.api_client)
        self.workspace_path = workspace_path

    def create_workspace_folder(self):
        """Create workspace folder if it doesn't exist."""
        try:
            self.workspace_api.mkdirs(self.workspace_path)
            logger.info(f"Created workspace folder: {self.workspace_path}")
        except Exception as e:
            logger.info(f"Folder may already exist: {str(e)}")

    def upload_notebook(self, local_path, workspace_name):
        """
        Upload a single notebook.

        Args:
            local_path: Local file path
            workspace_name: Name in workspace
        """
        try:
            # Read notebook content
            with open(local_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Encode content
            content_b64 = base64.b64encode(content.encode('utf-8')).decode('utf-8')

            # Determine format (Python or SQL)
            if local_path.endswith('.sql'):
                format_type = 'SQL'
            else:
                format_type = 'PYTHON'

            # Upload
            workspace_notebook_path = f"{self.workspace_path}/{workspace_name}"

            self.workspace_api.import_workspace(
                workspace_notebook_path,
                format=format_type,
                content=content_b64,
                overwrite=True
            )

            logger.info(f"✓ Uploaded: {workspace_name}")
            return True

        except Exception as e:
            logger.error(f"✗ Failed to upload {workspace_name}: {str(e)}")
            return False

    def deploy_all_notebooks(self, notebooks_dir):
        """
        Deploy all notebooks from directory.

        Args:
            notebooks_dir: Local notebooks directory path
        """
        logger.info("=" * 60)
        logger.info("DEPLOYING NOTEBOOKS TO DATABRICKS")
        logger.info("=" * 60)

        # Create workspace folder
        self.create_workspace_folder()

        # Get all notebook files
        notebooks_path = Path(notebooks_dir)
        notebook_files = list(notebooks_path.glob("*.py")) + list(notebooks_path.glob("*.sql"))

        if not notebook_files:
            logger.warning(f"No notebooks found in {notebooks_dir}")
            return

        logger.info(f"Found {len(notebook_files)} notebooks to deploy\n")

        # Upload each notebook
        success_count = 0
        for notebook_file in sorted(notebook_files):
            workspace_name = notebook_file.stem  # filename without extension
            if self.upload_notebook(str(notebook_file), workspace_name):
                success_count += 1

        logger.info("\n" + "=" * 60)
        logger.info(f"DEPLOYMENT COMPLETE: {success_count}/{len(notebook_files)} notebooks uploaded")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    # Get credentials from environment
    databricks_host = os.getenv("DATABRICKS_HOST")
    databricks_token = os.getenv("DATABRICKS_TOKEN")

    if not databricks_host or not databricks_token:
        logger.error("ERROR: DATABRICKS_HOST and DATABRICKS_TOKEN environment variables required")
        logger.error("Set them with:")
        logger.error("  export DATABRICKS_HOST=https://your-workspace.cloud.databricks.com")
        logger.error("  export DATABRICKS_TOKEN=your-personal-access-token")
        sys.exit(1)

    # Get notebooks directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    notebooks_dir = project_root / "notebooks"

    if not notebooks_dir.exists():
        logger.error(f"ERROR: Notebooks directory not found: {notebooks_dir}")
        sys.exit(1)

    # Deploy
    deployer = NotebookDeployer(
        host=databricks_host,
        token=databricks_token,
        workspace_path="/Medicaid/Notebooks"
    )

    deployer.deploy_all_notebooks(str(notebooks_dir))


if __name__ == "__main__":
    main()
