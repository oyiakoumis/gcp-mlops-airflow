import os
import json

import googleapiclient.discovery
from flask import Response


def make_prediction(request):
    """
    body example:
        {
          "instances": [
            [7, 0.27, 0.36, 20.7, 0.045, 45, 170, 1.001, 3, 0.45, 8.8]
          ]
        }
    """
    request_json = request.get_json()

    if (
        request.args
        and "key" in request.args
        and request.args["key"] == os.environ["API_KEY"]
    ):
        res = predict_json(
            "test-odysseas", "wine_quality", request_json["instances"]
        )
        return json.dumps(res)
    else:
        return Response(
            "Wrong authentication",
            status=401,
        )


def predict_json(project, model, instances, version=None):
    """
    Send json data to a deployed model for prediction.
    """
    service = googleapiclient.discovery.build("ml", "v1")
    name = "projects/{}/models/{}".format(project, model)

    if version is not None:
        name += "/versions/{}".format(version)

    response = (
        service.projects().predict(name=name, body={"instances": instances}).execute()
    )

    if "error" in response:
        raise RuntimeError(response["error"])

    return response["predictions"]
