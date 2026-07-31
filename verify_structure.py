import os

print("--- Checking Project Structure ---")
print(f"Current Directory: {os.getcwd()}")
print("Files in this directory:")
for item in os.listdir("."):
    print(f" - {item}")

print("\n--- Checking for main.py specifically ---")
if os.path.exists("main.py"):
    print("Found main.py!")
    try:
        # Let's try to just open and read the first few lines to make sure it's not a folder or empty
        with open("main.py", "r") as f:
            lines = f.readlines()
            if not lines:
                print("WARNING: main.py is empty!")
            else:
                print("main.py has content. First line:")
                print(f"  {lines[0].strip()}")
    except Exception as e:
        print(f"Error reading main.py: {e}")
else:
    print("CRITICAL ERROR: main.py is NOT in the current directory.")