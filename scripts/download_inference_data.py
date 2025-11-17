import zipfile
import requests
import urllib.parse
import os
import click


@click.command()
@click.option("--link", type=str, help="Yandex Disk link")
@click.option("--download_location", type=str, help="Download location")
def download_yandex_disk(link, download_location):
    url = f"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={link}"
    response = requests.get(url)
    download_url = response.json()["href"]
    file_name = urllib.parse.unquote(download_url.split("filename=")[1].split("&")[0])
    save_path = os.path.join(download_location, file_name)

    with open(save_path, "wb") as file:
        download_response = requests.get(download_url, stream=True)
        for chunk in download_response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
                file.flush()

    print("Download complete.")

    extract_zip(save_path)
    os.remove(save_path)

def extract_zip(zip_path: str):
    """Extract zip file."""
    if zipfile.is_zipfile(zip_path):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(".")
        print("Extraction complete!")
    else:
        print(f"{zip_path} is not a valid zip file")


if __name__ == "__main__":
    download_yandex_disk()