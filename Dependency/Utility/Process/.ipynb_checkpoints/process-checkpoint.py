import json
import time, sys
class Process_command():
    def __init__(self, command_file, section_name = "Process"):
        self.command_file = command_file
        self.section_name = section_name

    def execute(self):
        with open(self.command_file) as src:
            status = json.load(src)
        # self.last_index = status["State"]["last_index"]
        mode = status[self.section_name]["mode"]
        self.status = status
        match mode:
            case "none":
                return
            case "wait":
                wait_time = float(status[self.section_name]["wait_time"])
                time.sleep(wait_time)
                self.command_respond("none")
                
            case "stop":
                print("stopping")
                self.command_respond("exited")
                sys.exit()
            case "exited":
                self.command_respond("none")
            case "unrecognized input command":
                return
            case _:
                self.command_respond("unrecognized input command")
                
    def command_respond(self, message):
        self.status[self.section_name]["mode"] = message
        with open(self.command_file, "w") as dest:
            json.dump(self.status, dest, indent = 4)
