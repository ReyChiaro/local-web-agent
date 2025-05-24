from bs4 import BeautifulSoup


def parse_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    segs = []
    for p in soup.find_all(["p", "li", "h1", "h2", "h3", "h4"]):
        if len(p.get_text(strip=True)) > 25:
            segs.append(p)
    
    text = '\n'.join(segs)

    return text