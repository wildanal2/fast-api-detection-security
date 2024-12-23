def loger(df):
    # Display the HTML table
    html_table = df.head().to_html()

    # Create the HTML response
    return f"""
    <html>
        <head>
            <title>Debugging DataFrame</title>
        </head>
        <body>
            <h1>Data Debug</h1>
            {html_table}
        </body>
    </html>
    """
    