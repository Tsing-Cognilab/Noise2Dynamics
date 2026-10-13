"""Python 3.10 names needed by the pinned 2021 robosuite source."""
import collections
import collections.abc
for name in ('Iterable', 'Mapping', 'MutableMapping', 'Sequence'):
    if not hasattr(collections, name):
        setattr(collections, name, getattr(collections.abc, name))
