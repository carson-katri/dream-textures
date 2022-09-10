VERSION = (0, 0, 4)
def version_tag(version):
    return f"{version[0]}.{version[1]}.{version[2]}"

def version_tuple(tag):
    return tuple(tag.split('.'))