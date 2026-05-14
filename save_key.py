print("Paste your private key below.")
print("After pasting, press Enter then type DONE and press Enter.")
print("")

lines = []
while True:
    line = input()
    if line.strip() == "DONE":
        break
    lines.append(line)

key = "\n".join(lines) + "\n"

if "BEGIN RSA PRIVATE KEY" not in key:
    print("ERROR: That doesn't look like a private key. Try again.")
else:
    with open("kalshi_private.pem", "w") as f:
        f.write(key)
    print("Saved! Key length:", len(lines), "lines")
