import os
import shutil

OUTPUT_FOLDER = "output"
OUTPUT_CSV = "index_label_bbox.csv"

if __name__ == '__main__':
	if not os.path.isdir(OUTPUT_FOLDER):
		os.mkdir(OUTPUT_FOLDER)
		print("Created new folder to save images:", OUTPUT_FOLDER)
	print("Images will be copied to folder", OUTPUT_FOLDER)

	counter = 0

	with open(OUTPUT_CSV, 'x') as csv:
		csv.write("index,x,y,x_end,y_end\n")
		walk_dir = os.getcwd()
		for root, subdirs, files in os.walk(walk_dir):
			if OUTPUT_FOLDER in root:
				continue
			for f in files:
				if not f.endswith(".jpg"):
					continue
				try:
					box_x, box_y, end_x, end_y, _ = f[:-4].split('_')
				except:
					continue
				new_fname = f"{OUTPUT_FOLDER}/img{counter}.jpg"
				shutil.copy2(os.path.join(root, f), new_fname)
				csv.write(f"{new_fname},{box_x},{box_y},{end_x},{end_y}\n")
				counter += 1
	print("Done")