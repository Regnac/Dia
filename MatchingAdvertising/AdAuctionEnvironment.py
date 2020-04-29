from Environment import *


class AdAuctionEnvironment(Environment):
    def __init__(self, advertisers, publisher, users):
        self.advertisers = advertisers
        self.publisher = publisher
        self.users = users

    def simulate_users_behaviour(self, edges): #egdes are the edges of the superarm
        amount_of_clicks = np.zeros(len(edges))
        for i in range(len(self.users)):
            # TODO users select ad according to their features or some distribution
            clicked_ad_number = self.handle_clicked_ad_number(np.random.randint(len(edges)) + self.use_user_features(self.users[i]))
            amount_of_clicks[clicked_ad_number] += 1
        amount_of_clicks /= len(self.users)
        print("Amount of cliks ", amount_of_clicks)
        return amount_of_clicks

    def use_user_features(self, user):
        skip_ad = 0
        if(user.feature1 == 0):
            skip_ad += 1
        if(user.feature1 == 1):
            skip_ad += -1
        if (user.feature2 == 0):
            skip_ad += 2
        if (user.feature2 == 1):
            skip_ad += -2

        return skip_ad

    def handle_clicked_ad_number(self,clicked_ad_number): #TAKE CARE OF SOME ARRAY INDEX OUT OF BOUND
        if clicked_ad_number > 3:
            clicked_ad_number -= 3
        if clicked_ad_number < 0:
            clicked_ad_number -= clicked_ad_number

        return clicked_ad_number