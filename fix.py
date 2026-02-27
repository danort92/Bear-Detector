import re, ast

with open("apply_refactor.py") as f:
    content = f.read()

# Trova tutti i pezzi di stringa dentro DATA = (...)
pieces = re.findall(r"'([^']*)'", content)
joined = "".join(pieces)

new_script = f'''import base64, io, os, zipfile
DATA = "{joined}"
def main():
    buf = io.BytesIO(base64.b64decode(DATA))
    with zipfile.ZipFile(buf, "r") as zf:
        for name in zf.namelist():
            os.makedirs(os.path.dirname(name) or ".", exist_ok=True)
            zf.extract(name)
            print("  wrote:", name)
    old = "Bear_detection.ipynb"
    if os.path.exists(old):
        os.remove(old)
        print("  deleted:", old)
    print("\\nDone!")
if __name__ == "__main__":
    main()
'''

with open("apply_refactor_fixed.py", "w") as f:
    f.write(new_script)

print("Creato apply_refactor_fixed.py - ora esegui: python apply_refactor_fixed.py")
