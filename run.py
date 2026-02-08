from app import create_app

app = create_app()

if __name__ == '__main__':
    # Debug mode allows live reloading when you change code
    app.run(debug=True, port=5000)