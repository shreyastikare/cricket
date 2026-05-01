import ipl

def train_all():
    ipl.resource_train_partial()
    ipl.first_innings_glm_train_partial()
    ipl.resource_train_full()
    ipl.first_innings_glm_train_full()
    ipl.second_innings_glm_train_partial()
    ipl.second_innings_glm_train_full()
    return