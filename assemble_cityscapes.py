import os

from cityscapes import CityscapesAssembler
from six.moves import input


def main():
  folder = "/ais/gobi4/mren/data/cityscapes"
  for split in ["train", "valid", "test"]:
    output_fname = os.path.join(folder, "sem_seg",
                                "{}_full_size.h5".format(split))
    if os.path.exists(output_fname):
      confirm = input("Overwrite existing file {}? [Y/n]".format(output_fname))
      if confirm == "n" or confirm == "N":
        return
    a = CityscapesAssembler(
        folder=folder,
        height=-1,
        width=-1,
        split=split,
        coarse_label=False,
        output_fname=output_fname,
        semantic_only=True)
    a.assemble()


if __name__ == "__main__":
  main()
