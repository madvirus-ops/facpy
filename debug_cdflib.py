
import cdflib
print(f"Version: {cdflib.__version__}")
try:
    print(f"cdfepoch: {cdflib.cdfepoch}")
except AttributeError:
    print("cdflib.cdfepoch not found")

try:
    print(f"cdfepoch.tt2000_to_datetime: {cdflib.cdfepoch.tt2000_to_datetime}")
except AttributeError:
    print("tt2000_to_datetime not found")

try:
    print(f"cdfepoch.to_datetime: {cdflib.cdfepoch.to_datetime}")
except AttributeError:
    print("to_datetime not found")
