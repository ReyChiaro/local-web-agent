from scripts.build_vectorstore import build_vectorstore, load_urls_from_file
from app.run_chat import run_chat


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WebAgent")
    parser.add_argument("urls", nargs="*", help="List of URLs to be processed.",default=["https://reychiaro.github.io/about"])
    parser.add_argument("--url-file", help="Path to file containing list of URLs to be processed.",)

    args = parser.parse_args()

    urls = []
    if args.url_file:
        urls.extend(load_urls_from_file(args.url_file))
    urls.extend(args.urls)

    if not urls:
        raise RuntimeError("No URLs to be processed, check your passed args.\nUsage: python start.py <URL1> <URL2> ... OR path/to/urls.txt")
    build_vectorstore(urls)
    run_chat()


