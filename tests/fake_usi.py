"""Faux moteur USI pour les tests : suit le protocole et joue toujours 7g7f."""
import sys


def main():
    multipv = 1
    for line in sys.stdin:
        cmd = line.strip()
        if cmd == "usi":
            print("id name FauxMoteur 1.0")
            print("option name MultiPV type spin default 1 min 1 max 10")
            print("usiok", flush=True)
        elif cmd == "isready":
            print("readyok", flush=True)
        elif cmd.startswith("setoption name MultiPV"):
            multipv = int(cmd.split()[-1])
        elif cmd.startswith("go"):
            print("info depth 1 multipv 1 score cp 10 pv 2g2f")
            print("info depth 8 multipv 1 score cp 55 pv 7g7f 3c3d 2g2f")
            if multipv >= 2:
                print("info depth 8 multipv 2 score mate -3 pv 2g2f 8c8d")
            print("bestmove 7g7f ponder 3c3d", flush=True)
        elif cmd == "quit":
            return


main()
