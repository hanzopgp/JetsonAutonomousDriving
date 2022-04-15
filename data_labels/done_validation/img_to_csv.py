import os
import shutil

OUTPUT_FOLDER = "output"
OUTPUT_CSV = "index_label.csv"

LABELS = ["FIST", "FINGER", "PALM", "LEFT", "RIGHT"]

if __name__ == '__main__':
	if not os.path.isdir(OUTPUT_FOLDER):
		os.mkdir(OUTPUT_FOLDER)
		print("Created new folder to save images:", OUTPUT_FOLDER)
	print("Images will be copied to folder", OUTPUT_FOLDER)

	counter = {label:0 for label in LABELS}

	with open(OUTPUT_CSV, 'x') as csv:
		csv.write("index,label\n")
		walk_dir = os.getcwd()
		for root, subdirs, files in os.walk(walk_dir):
			for f in files:
				if not f.endswith(".jpg"):
					continue
				for lb in LABELS:
					if f.startswith(lb):
						new_fname = f"{OUTPUT_FOLDER}/{lb}_{counter[lb]}.jpg"
						shutil.copy2(os.path.join(root, f), new_fname)
						csv.write(f"{new_fname},{lb}\n")
						counter[lb] += 1