from pathlib import Path

DATASET_PATH = Path("data/bbc")

categories = [
	"business",
	"entertainment",
	"politics",
	"sport",
	"tech"
]

total_documents = 0

print("BBC DATASET VERIFICATION")
print("=" * 40)

for category in categories:

	folder = DATASET_PATH / category

	if not folder.exists():
		print(f"{category}: Folder NOT FOUND")
		continue

	files = list(folder.glob("*.txt"))

	count = len(files)
	total_documents += count

	print(f"{category}: {count} documents")

print("=" * 40)
print("Total documents:", total_documents)

if total_documents >= 75:
	print("Dataset size requirement: PASSED")
else:
	print("Dataset size requirement: NOT PASSED")
