import argparse, shutil, zipfile, os


def get_data(args):
    source = args.data_path
    destination = '../data/raw/Images.zip'

    print('Copying image files...')
    shutil.copy(source, destination)

    with zipfile.ZipFile(source, 'r') as zip_ref:
        zip_ref.extractall('../data/raw')

    os.remove(destination)
    print('{} images extracted'.format(len(os.listdir('../data/raw/Images'))))
    print('Images copied and extracted to:','../data/raw')
    print("Done!")

if __name__ == '__main__':
    # Obtain image data path from the Command Line Interface (CLI)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", "-dp", type=str, dest="data_path",
        required=True, help="script of model to run."
        )
    args = parser.parse_args()

    get_data(args)