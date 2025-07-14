import socket
import pickle
import random
import importlib

aes=importlib.import_module("2005102_aes")
keyExpansion=aes.keyExpansion
aesDecryptCBC=aes.aesDecryptCBC
unpad=aes.unpad

ecc=importlib.import_module("2005102_ecc")
EllipticCurve=ecc.EllipticCurve


server=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
server.bind(('localhost',1234))
server.listen(5)
print("Server is running and waiting for clients...")


while True:
    print("To close server press 'q' and to continue press 'c")
    typedChar=input()
    if(typedChar=='q'):
        break
    try:
        connection,address=server.accept()
        print(f"Server is connected to {address}")

        startMsg=connection.recv((4096)).decode()
        # if(startMsg!="I want to send you a encrypted message"):
        #     continue
        print(f"[From Client]-{startMsg}")
        response=input("Your reply (y/n):")
        connection.send(response.encode())
        if(response=='n'):
            continue

        data=pickle.loads(connection.recv(4096))

        a=data['a']
        b=data['b']
        p=data['p']
        G=data['G']

        #Client's public key
        A=data['A']

        curve=EllipticCurve(a,b,p)

        # print(f"Server received client's public key: {A}")

        #Generate Server's private and public key
        Kb=random.getrandbits(128)
        B=curve.doubleAndAdd(Kb,G)

        #Send public key to client
        # print(f"Server is sending server's public key: {B}")
        connection.send(pickle.dumps(B))

        #Compute shared secret key
        R=curve.doubleAndAdd(Kb,A)
        sharedKey=R[0].to_bytes(16,byteorder='big')
        print("Encryption key is ready")

        #Inform Client key is ready
        connection.send("Encryption Key Generated.If you are ready to transfer type 'y' else 'n'".encode())


        # connection.send("If you are ready to transfer type 'y' else 'n'".encode())


        #Waiting for client's confirmation
        clientConfMsg=connection.recv(4096).decode()
        if clientConfMsg != "y":
            print("Client is not ready yet.Waiting for others to connect...")
            connection.close()
            continue

        print("Client is ready!")

        #Receive encrypted payload
        payload=pickle.loads(connection.recv(4096))
        ciphertext=payload['cipher']
        Rcon=payload['rcon']
        # print(f"From server: {''.join([chr(byte) for byte in ciphertext])}")

        #Decrypt message
        roundKeys=keyExpansion(sharedKey,Rcon)
        plaintext=aesDecryptCBC(ciphertext,sharedKey,roundKeys)
        plaintext=unpad(plaintext)
        print(f"\nDecrypted message: {plaintext.decode()}")

    except Exception as e:
        print(f"Error during communication: {e}")

    finally:
        connection.close()
        print("Connection closed.Waiting for new clients...\n")