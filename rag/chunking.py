from rag.parsing import parse_docs, DOCS_DIR
import re

CHUNK_BUDGET = 500


class Chunk:
    def __init__(self, text, breadcrumb, source, file_path):
        self.text = text
        self.breadcrumb = breadcrumb
        self.source = source
        self.file_path = file_path


def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def chunk_section(section):
    sentences = split_into_sentences(section.text)
    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) + 1 > CHUNK_BUDGET and current:
            chunks.append(
                Chunk(" > ".join(section.breadcrumb) + " : " + " ".join(current), section.breadcrumb, section.source, section.file_path)
            )
            current = []
            current_len = 0
        
        current.append(sentence)
        current_len += len(sentence) + 1
    
    if current:
        chunks.append(
            Chunk(" > ".join(section.breadcrumb) + " : " + " ".join(current), section.breadcrumb, section.source, section.file_path)
        )

    return chunks


def chunk_sections(sections):
    chunks = []
    for section in sections:
        chunks.extend(chunk_section(section))
    return chunks


if __name__ == "__main__":
    wiki_sections = parse_docs(DOCS_DIR / "wiki")
    manual_sections = parse_docs(DOCS_DIR / "manual")
    wiki_chunks = chunk_sections(wiki_sections)
    manual_chunks = chunk_sections(manual_sections)
    print(f"Wiki chunks: {len(wiki_chunks)}")
    print(f"Manual chunks: {len(manual_chunks)}")
    print(f"Total chunks: {len(wiki_chunks) + len(manual_chunks)}")
