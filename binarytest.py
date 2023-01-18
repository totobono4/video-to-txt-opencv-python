import binascii

print("{:04X}".format(65535))
print(binascii.unhexlify("{:04X}".format(65535)))

print(binascii.unhexlify('0168'))
print(binascii.unhexlify('01E0'))

print(binascii.hexlify(b'\x01h'))
print(binascii.hexlify(b'\x01\xe0'))
