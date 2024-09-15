import warnings
from pipeline import Pipleline
from flask import Flask, request, jsonify

warnings.filterwarnings("ignore")

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def processCSVFile ():

    try :

        csv_file = request.json.get('file').get('content')
        query = request.json.get('query')

        if csv_file is None or query is None:
            print(400)
            return jsonify({
                'status': 400,
                'message': "The necessary parameters must be provided"
            })
    
    except Exception as e:
        return jsonify({
            'status': 400,
            'message': f"Error in link configuration, {e}"
        })
    
    try:

        PL = Pipleline(query, csv_file)
        response, results = PL.forward()

    except Exception as e:
        
        return jsonify({
            'status': 500,
            'message': "Internal server error",
        })
    
    return jsonify({
        'status': 200,
        'message': 'Success',
        'response': response,
        'results': results
    })

if __name__ == '__main__':
    app.run(debug=True)
