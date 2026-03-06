import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

posts = json.load(open("se_posts_normalized.json", "r", encoding="utf-8"))
post_text = {}
for thread in posts:
    for p in thread["posts"]:
        post_text[p["id"]] = p["text"]

# Show full text of the 5 skipped questions
skipped = [
    "se-engineering-55872",   # AWJC question
    "se-engineering-54734",   # retaining ring question
    "se-engineering-51190",   # turning calculation question
    "se-engineering-51401",   # desktop lathe question
    "se-engineering-51293",   # milling tool question
    "se-engineering-49506",   # vinyl record hardness question
]

for pid in skipped:
    txt = post_text.get(pid, "NOT FOUND")
    print(f"\n{'='*80}")
    print(f"POST: {pid}")
    print(f"{'='*80}")
    print(txt)
    print()
