import os

print(__file__)
print(os.path.realpath(os.path.relpath(__file__)))
print(os.path.relpath(__file__))