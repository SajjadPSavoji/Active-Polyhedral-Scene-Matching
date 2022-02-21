import imaplib
import email
from email.header import decode_header
import webbrowser
import os
from tqdm import tqdm

# account credentials
username = "__Your Gmail ID__"
password = "__Your Pass Word__"
rcv_email = "polyhedral@eecs.yorku.ca"
# number of top emails to fetch
N = 10 
# ids will be stored in IDs
IDs = []
# ids will be saved in the destination_path in txt format
destination_path = "./Object_IDs.txt"

def clean(text):
    # clean text for creating a folder
    return "".join(c if c.isalnum() else "_" for c in text)


# create an IMAP4 class with SSL 
imap = imaplib.IMAP4_SSL("imap.gmail.com")
# authenticate
imap.login(username, password)
# select Inbox
status, messages = imap.select("INBOX")
# total number of emails
messages = int(messages[0])

for i in tqdm(range(messages, messages-N, -1)):
    # fetch the email message by ID
    res, msg = imap.fetch(str(i), "(RFC822)")
    for response in msg:
        if isinstance(response, tuple):
            # parse a bytes email into a message object
            msg = email.message_from_bytes(response[1])
            # decode the email subject
            subject, encoding = decode_header(msg["Subject"])[0]
            if isinstance(subject, bytes):
                # if it's a bytes, decode to str
                subject = subject.decode(encoding)
            # decode email sender
            From, encoding = decode_header(msg.get("From"))[0]
            if isinstance(From, bytes):
                From = From.decode(encoding)
            # print("Subject:", subject)
            # print("From:", From)
            if not From == rcv_email:
                continue
            # if the email message is multipart
            if msg.is_multipart():
                # iterate over email parts
                for part in msg.walk():
                    # extract content type of email
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    try:
                        # get the email body
                        body = part.get_payload(decode=True).decode()
                    except:
                        pass
                    if content_type == "text/plain" and "attachment" not in content_disposition:
                        # parse the message to get the ID
                        IDs.append(body.split('API: ')[1].split('\r')[0])
                    elif "attachment" in content_disposition:
                        # download attachment
                        filename = part.get_filename()
                        if filename:
                            folder_name = clean(subject)
                            if not os.path.isdir(folder_name):
                                # make a folder for this email (named after the subject)
                                os.mkdir(folder_name)
                            filepath = os.path.join(folder_name, filename)
                            # download attachment and save it
                            open(filepath, "wb").write(part.get_payload(decode=True))
            else:
                # extract content type of email
                content_type = msg.get_content_type()
                # get the email body
                body = msg.get_payload(decode=True).decode()
                if content_type == "text/plain":
                    # print only text email parts
                    print(body)

# write IDs to the destination path
with open(destination_path, 'w') as fp:
    for i in IDs:
        print(i, file=fp)

imap.close()
imap.logout()