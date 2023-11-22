import json
import sys
from flask_restful import Resource, Api
import greenlet
from sqlalchemy.orm import scoped_session
from db import SessionLocal, Project, ProjectData
from flask import Flask, request, abort
from flask_cors import CORS

app = Flask(__name__)
rest = Api(app)
CORS(app) # resources={r'/*': {'origins': '*'}}

app.db_session = scoped_session(SessionLocal, scopefunc=greenlet.getcurrent)


@app.teardown_appcontext
def remove_session(*args, **kwargs):
    app.db_session.remove()


@app.route('/')
def hello_world():
    return 'Hello, World!'  # Maybe serve the the compiled webapp to simplify the demo? /Mattias


def _box_project(proj):
    p = proj.to_dict()
    p['props'] = json.loads(p['props']) if 'props' in p and p['props'] is not None else None
    return p


def _box_data(data):
    _d = data.to_dict()
    print(_d, flush=True)
    d = json.loads(_d['data'])
    d['element'] = {
        "pid": _d['pid'],
        "key": _d['key'] if 'key' in _d else None,
        "version": _d['skey'] if 'skey' in _d else None,
        "dt": -1 if _d['dt'] is None else _d['dt'].timestamp()
    }
    return d


class ProjectResource(Resource):
    def get(self, pid):
        project = app.db_session.query(Project).filter(Project.id == pid).first()
        if project is None:
            return {}

        out = _box_project(project)

        if 'data' in request.args and request.args['data'] == 'true':
            data = app.db_session.query(ProjectData).filter(ProjectData.pid == pid).all()
            data = [_box_data(d) for d in data]

            out['data'] = {}
            for d in data:
                if d['element']['key'] not in out['data']:
                    out['data'][d['element']['key']] = []
                out['data'][d['element']['key']].append(d)
        return out

    def put(self, pid):
        r = request.json
        props = json.dumps(r['props'])
        project = Project(id=pid, props=props)
        app.db_session.add(project)
        app.db_session.commit()
        print(pid, props, flush=True)
        return {}


class ProjectListResource(Resource):
    def get(self):
        projects = app.db_session.query(Project).all()
        return {'projects': [_box_project(p) for p in projects]}


class ProjectDataListResource(Resource):

    def get(self, pid, key, skey):
        if key is None:
            data = app.db_session.query(ProjectData).filter(ProjectData.pid == pid).all()
        elif skey is None:
            data = app.db_session.query(ProjectData).filter(ProjectData.pid == pid, ProjectData.key == key).all()
        else:
            data = app.db_session.query(ProjectData).filter(ProjectData.pid == pid, ProjectData.key == key, ProjectData.skey == skey).all()

        return {'data': [_box_data(d) for d in data]}

    def put(self, pid, key, skey):
        d = request.json
        print("put data: ", pid, key, skey, json.dumps(d), file=sys.stdout, flush=True)
        data = ProjectData(pid=pid, key=key, skey=skey, data=json.dumps(d))
        app.db_session.merge(data)
        app.db_session.commit()
        return {}

    def post(self, pid, key, skey):
        d = request.json
        if key is None:
            abort(400)
        data = ProjectData(pid=pid, key=key, skey=skey, data=json.dumps(d))
        app.db_session.add(data)
        app.db_session.commit()
        return {}


class ProjectDataResource(Resource):

    def get(self, pid, key, skey):
        if key is None:
            data = app.db_session.query(ProjectData).filter(ProjectData.pid == pid).first()
        elif skey is None:
            data = app.db_session.query(ProjectData).filter(ProjectData.pid == pid, ProjectData.key == key).first()
        else:
            data = app.db_session.query(ProjectData).filter(ProjectData.pid == pid, ProjectData.key == key, ProjectData.skey == skey).first()

        return _box_data(data)

    def put(self, pid, key, skey):
        d = request.json
        print("put data: ", pid, key, skey, json.dumps(d), file=sys.stdout, flush=True)
        data = ProjectData(pid=pid, key=key, skey=skey, data=json.dumps(d))
        app.db_session.merge(data)
        app.db_session.commit()
        return {}

    def post(self, pid, key, skey):
        d = request.json
        if key is None:
            abort(400)
        data = ProjectData(pid=pid, key=key, skey=skey, data=json.dumps(d))
        app.db_session.add(data)
        app.db_session.commit()
        return {}


rest.add_resource(ProjectListResource, "/project")
rest.add_resource(ProjectResource, "/project/<string:pid>")
rest.add_resource(ProjectDataListResource, "/project/<string:pid>/data/", defaults={'key': None, 'skey': None}, endpoint='project_data_1')
rest.add_resource(ProjectDataListResource, "/project/<string:pid>/data/<string:key>", defaults={'skey': None}, endpoint='project_data_2')
rest.add_resource(ProjectDataResource, "/project/<string:pid>/data/<string:key>/<string:skey>", endpoint='project_data_3')

# rest.add_resource(ProjectDataEntryResource, "/project/<string:pid>/<string:data>/<string:key>")

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=6060, threaded=True)