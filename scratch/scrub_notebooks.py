import json
import glob
import os

token_to_replace = "hf_" + "ppJDsfztCoryazllpRVIbxWLtZITkWUztd"
replacement = "import os\\nHF_TOKEN = os.environ.get(\\\"HF_TOKEN\\\", \\\"YOUR_HF_TOKEN\\\")"

for path in glob.glob("d:/X/notebooks/*.ipynb"):
    print(f"Scrubbing {path}...")
    with open(path, "r", encoding="utf-8") as f:
        try:
            nb = json.load(f)
        except Exception as e:
            print(f"Failed to parse {path}: {e}")
            continue
            
    modified = False
    for cell in nb.get("cells", []):
        if "source" in cell:
            new_source = []
            for line in cell["source"]:
                if token_to_replace in line:
                    print(f"  Found token in cell: {line.strip()}")
                    # Replace the hardcoded assignment line with the env var version
                    # If the line is 'HF_TOKEN = "hf_..."\n', we want to replace it
                    # Let's replace the token specifically or the whole line
                    line = line.replace(f'"{token_to_replace}"', 'os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN")')
                    line = line.replace(f'\\"{token_to_replace}\\"', 'os.environ.get(\\"HF_TOKEN\\", \\"YOUR_HF_TOKEN\\")')
                    line = line.replace(token_to_replace, 'YOUR_HF_TOKEN')
                    # Also make sure os is imported
                    if "import os" not in line and not any("import os" in l for l in cell["source"]):
                        # Inject import os at the start of the cell or on the line
                        line = "import os\n" + line
                    modified = True
                    print(f"  Replaced with: {line.strip()}")
                new_source.append(line)
            cell["source"] = new_source
            
    if modified:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Saved {path}")
