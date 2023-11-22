# Integration demo

This file demonstrates interconnection between the functionality of the aequitas-lib and the context-service.
The context-service is the connection point between the different tools of the AEQUITAS project holding information about
current projects and their elements.

This interconnection is handled by the aequitas.gateway.
In order to test this interconnection, a context-service needs to be present and running.
A mockup context service placeholder is provided in: https://github.com/handsondemo/prototype-api

### Usage:
* clone git repo
* (create venv)
* pip install -r requirements
* python aeq-api/server.py

This will start a context-service mockup on http://localhost:6060

### Running the examples without the context-service

The gateway also provides a file-system mode by passing filesystem=True to the gateway.
This will persist element data to a filesystem structure and json-files instead of using the context service.