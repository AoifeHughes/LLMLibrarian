import re 


def remove_markdown_syntax(markdown_text):
    plain_text = re.sub(r'^#.*$', '', markdown_text, flags=re.MULTILINE)
    plain_text = re.sub(r'(\*{1,2}|_{1,2})(.*?)\1', r'\2', plain_text)
    plain_text = re.sub(r'\[.*?\]\(.*?\)', '', plain_text)
    plain_text = re.sub(r'!\[.*?\]\(.*?\)', '', plain_text)
    plain_text = re.sub(r'```[^`]*```', '', plain_text, flags=re.DOTALL)
    plain_text = re.sub(r'`[^`]*`', '', plain_text)
    plain_text = re.sub(r'^\s*>\s*', '', plain_text, flags=re.MULTILINE)
    plain_text = re.sub(r'^[-*_]{3,}$', '', plain_text, flags=re.MULTILINE)
    plain_text = re.sub(r'^\|.*\|$', '', plain_text, flags=re.MULTILINE)
    plain_text = re.sub(r'\[\^.*?\]', '', plain_text)
    plain_text = re.sub(r'^\[.*?\]:.*$', '', plain_text, flags=re.MULTILINE)
    plain_text = re.sub(r'<[^>]*>', '', plain_text)
    return plain_text.strip()