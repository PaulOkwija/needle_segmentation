import argparse, shutil, zipfile, os


def get_data(args):
    source = args.data_path
    destination = '../data/raw/Images.zip'

    csv_files = [f for f in os.listdir(source) if f.endswith('.csv')]
    if len(csv_files) == 0:
        raise ValueError('No CSV files found in the source directory')
    image_path = source + '/Images.zip'

    print('Copying image files...')
    shutil.copy(image_path, destination)

    for csv in csv_files:
        shutil.copy(source + '/' + csv, '../data/raw')

    with zipfile.ZipFile(image_path, 'r') as zip_ref:
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