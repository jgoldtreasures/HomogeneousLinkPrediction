def text(body):
    import smtplib
    email = "pythonsmsjulian@gmail.com"
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(email, "JulianPythonSMS")
    # body = "All Done\n"
    server.sendmail(email, '7733267749@vtext.com', body)
    # server.sendmail(email, '3108953872@tmomail.net', body)
    server.quit()


text("test")
