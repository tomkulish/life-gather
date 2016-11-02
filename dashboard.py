# The dashboard of tomkulish.com life-gather

import pymongo
import bottle

__author__ = 'tbk'

@bottle.route('/')
def blog_index():

    cookie = bottle.request.get_cookie("session")

    username = sessions.get_username(cookie)

    # todo: this is not yet implemented at this point in the course

    #return bottle.template('blog_template', dict(username=username))
