import os

print("Running all tools on ../../TestSet")
print("Please allow a few minutes")
print()

print("Progpilot")
os.system("cd progpilot && python3 run_comparison.py")
print()
print("RIPS")
os.system("cd rips && python3 run_comparison.py")
print()
print("TAP")
os.system("cd TAP && python3 run_comparison.py")
print()
print("WAP ... this takes a lot longer to run than the other tools. Please allow a few minutes")
os.system("cd wap && python3 run_comparison.py")