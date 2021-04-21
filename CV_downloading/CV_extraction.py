import smtplib
import mimetypes
import imaplib
import base64
import os
import email
import traceback

import sys
import email.header
import datetime

from email.mime.multipart import MIMEMultipart
from email import encoders
from email.message import Message
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

from CV_extraction_config import SERVER, USER, PASSWORD, SAVED_DIRECTORY, NB_LAST_MSG

def connect(server, user, password):
    """Connexion à la boite mail"""
    connexion = imaplib.IMAP4_SSL(server)
    print("Connexion for USER : ", user)
    connexion.login(user, password)
    connexion.select()
    return connexion


def close_session(connexion):
    connexion.close()
    connexion.logout()
    print("Deconnexion : Success")
    

def downloaAttachmentsInEmail(connexion, emailid, outputdir):
    try:
        #si emailid est type byte
        resp, data = connexion.fetch(emailid, "(BODY.PEEK[])")
    except:
        #si emailid est type string
        resp, data = connexion.fetch(str.encode(emailid), "(BODY.PEEK[])")
    email_body = data[0][1]
    mail = email.message_from_bytes(email_body)
    if mail.get_content_maintype() != 'multipart':
        return
    for part in mail.walk():
        if part.get_content_maintype() != 'multipart' and part.get('Content-Disposition') is not None:
            open(outputdir + '/' + part.get_filename(), 'wb').write(part.get_payload(decode=True))


def subjectQuery(connexion, subject, outputdir):
    """Input : Objet du mail
       Output : Telecharge en local la PJ"""
    connexion.select("Inbox")
    typ, msgs = connexion.search(None, '(SUBJECT "' + subject + '")')
    msgs = msgs[0].split()
    for emailid in msgs:
        print(emailid)
        downloaAttachmentsInEmail(connexion, emailid, outputdir)


def get_last_mails(connexion, x):
    """Input: connexion et x derniers ids de mails que l'on souhaite récupérer
       Outputs: liste de x derniers ids des mails"""
    # On récupère la liste de tous les index des mails de INBOX (à confirmer pour "tous")
    rv, data = connexion.search(None, "ALL")
    if rv != 'OK':
        print("No messages found!")
    #Binary to string + clean
    index_list = data[0].decode("utf-8").split()
    index_list.reverse()
    print("Scrapping des :", x, " CV les plus récents")
    return index_list[:x]


def get_subject(connexion, list_mails_index):
    """Input: connexion et nbe de mails que l'on souhaite récupérer
       Output: """

    connexion.select('INBOX', readonly=True)
    from_list, subject_list = [], []
    for i in list_mails_index:
        typ, msg_data = m.fetch(str(i), '(RFC822)')
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_string(response_part[1].decode("utf-8"))
                for header in [ 'subject', 'to', 'from' ]:
                    #print('%-8s: %s' % (header.upper(), msg[header]))
                    if header == 'subject':
                        subject_list.append(msg[header])
                    if header == 'from':
                        from_list.append(msg[header])
    return from_list, subject_list


if __name__ == "__main__":
    #connexion
    connexion = connect(SERVER, USER, PASSWORD)
    try:
        #get list of last 5 mails
        list_mails_ids = get_last_mails(connexion, NB_LAST_MSG)
        # get from and subject (not mandatory)
        #fr, sub = get_subject(connexion, list_mails_ids)
        #TEST (on telecharge les PJ de la liste de mails p)
        for id_ in list_mails_ids:
            downloaAttachmentsInEmail(connexion, id_, SAVED_DIRECTORY)
        close_session(connexion)
    except:
        traceback.print_exc()
        close_session(connexion)

