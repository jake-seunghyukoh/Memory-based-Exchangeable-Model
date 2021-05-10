def printInfo(x):
    print(f"[INFO]:  {x}")


def printError(x):
    print(f"[ERROR]: {x}")


def printDebug(x, debug):
    if debug:
        print(f"[DEBUG]: {x}")
