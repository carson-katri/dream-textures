VERSION = (0, 0, 6)
def version_tag(version):
    return f"{version[0]}.{version[1]}.{version[2]}"

def version_tuple(tag):
    return tuple(int(tag.split('.')))
