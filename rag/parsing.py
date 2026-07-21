from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent / "docs"


class Section:
    def __init__(self, file_path, breadcrumb, text, source):
        self.file_path = file_path
        self.breadcrumb = breadcrumb
        self.text = text
        self.source = source


def parse_file(file_path):
    sections = []
    stack = []
    current_lines = []

    with open(file_path) as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("#"):
            if stack:
                text = "".join(current_lines).strip()
                breadcrumb = [title for level, title in stack]
                source = file_path.parent.name
                sections.append(Section(file_path, breadcrumb, text, source))
            current_lines = []

            level = len(line) - len(line.lstrip("#"))
            title = line.strip("#").strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            current_lines.append(line)

    if stack:
        text = "".join(current_lines).strip()
        breadcrumb = [title for level, title in stack]
        source = file_path.parent.name
        sections.append(Section(file_path, breadcrumb, text, source))

    return sections


def parse_docs(folder):
    sections = []
    for file_path in sorted(Path(folder).glob("*.md")):
        sections.extend(parse_file(file_path))
    return sections


if __name__ == "__main__":
    wiki_sections = parse_docs(DOCS_DIR / "wiki")
    manual_sections = parse_docs(DOCS_DIR / "manual")

    # print("=== wiki ===")
    # for section in wiki_sections[:5]:
    #     print(section.file_path)
    #     print(section.breadcrumb)
    #     print(section.text)
    #     print()

    # print("=== manual ===")
    # for section in manual_sections[:5]:
    #     print(section.file_path)
    #     print(section.breadcrumb)
    #     print(section.text)
    #     print(section.source)
    #     print()
