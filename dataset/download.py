########################################################################

import sys
import os
import urllib.request
import tarfile
import zipfile

########################################################################


def _print_download_progress(count, block_size, total_size):
    """
    Function that prints the download progress.
    """

    pct_complete = float(count * block_size) / total_size

    msg = "\r- Download progress: {0:.1%}".format(pct_complete)

    sys.stdout.write(msg)
    sys.stdout.flush()


########################################################################


def maybe_download_and_extract(url, download_dir):
    """
    Dowload and extract the data
    :param url: URL from tar archive to be downloaded.
    :param download_dir: Directory where the archive will be saved.
    """

    # Archive name
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)

    # Verify the archive existance
    if not os.path.exists(file_path):
        os.makedirs(download_dir, exist_ok=True)

        # Download the archive
        file_path, _ = urllib.request.urlretrieve(url=url,
                                                  filename=file_path,
                                                  reporthook=_print_download_progress)

        print()
        print("Downloaded! Extracting...")

        if file_path.endswith(".zip"):
            # Descompact zip archive
            with zipfile.ZipFile(file=file_path, mode="r") as zip_ref:
                zip_ref.extractall(download_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            # Descompact tar-ball.
            with tarfile.open(name=file_path, mode="r:gz") as tar_ref:
                tar_ref.extractall(download_dir)

        print("Success.")
    else:
        print("Data is already downloaded.")


########################################################################
