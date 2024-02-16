VERSION = (0, 3, 1)
def version_tag(version):
    return f"{version[0]}.{version[1]}.{version[2]}"

def version_tuple(tag):
    return tuple(map(lambda x: int(x), tag.split('.')))
