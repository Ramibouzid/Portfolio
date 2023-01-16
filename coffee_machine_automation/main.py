from data import MENU, resources
profit = 0


def process_coin():
    print("Please insert coins.")
    total = int(input("how many quarters?:"))*0.25
    total += int(input("how many dimes?:"))*0.1
    total += int(input("how many nickles?:"))*0.05
    total += int(input("how many pennies?:"))*0.01
    return total


def make_coffee():
    for i in resources:
        resources[i] -= MENU[choice]["ingredients"][i]


def report():
    print("water:", resources["water"], "ml")
    print("milk:", resources["milk"], "ml")
    print("coffee:", resources["coffee"], "g")
    print(f"Money:, {profit}", "$")


end_game = False
while not end_game:
    choice = input("What would you like? (espresso/latte/cappuccino):").lower()

    if choice == "off":
        end_game = True
    elif choice == "report":
        report()
    else:
        drink = MENU[choice]
        if resources["water"] < drink["ingredients"]["water"] or resources["milk"] < drink["ingredients"]["milk"] or resources["coffee"] < drink["ingredients"]["coffee"]:
            print("not enough resources to make the coffee 😭😭")
            end_game = True
        else:
            pay = process_coin()
            refund = round(pay - drink["cost"], 2)
            if pay < drink["cost"]:
                print("sorry, not enough money. Money refunded.")
                print(f"Here is ${refund} in change.")
            else:
                make_coffee()
                profit += drink["cost"]
                print(f"Here is ${refund} in change.")
                print("Here is your latte ☕️. Enjoy!")
