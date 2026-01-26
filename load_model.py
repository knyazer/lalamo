from huggingface_hub import snapshot_download

snapshot_download(repo_id="mlx-community/gemma-3-4b-it-8bit", local_dir="test-model", local_dir_use_symlinks=False)
