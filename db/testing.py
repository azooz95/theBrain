from collection_Control import (
    register_user,
    login_user,
    insert_document,
    search_by_filename,
    delete_document
)

print("\n🔐 Registering user...")
res = register_user("Sarah", "sarah@example.com", "securepass123")
print(res.get("message", "No message returned."))

print("\n🔐 Trying to register with duplicate email...")
res = register_user("Another", "sarah@example.com", "newpass")
print(res.get("message", "No message returned."))

print("\n🔐 Logging in with wrong password...")
login_result = login_user("sarah@example.com", "wrongpass")
if not login_result or not login_result.get("success", False):
    print(login_result.get("message", "❌ Login failed (as expected)."))

print("\n🔐 Logging in with correct password...")
user_data = login_user("sarah@example.com", "securepass123")

if not user_data.get("success"):
    print(user_data.get("message", "❌ Login failed unexpectedly."))
else:
    print(f"✅ Login successful for {user_data['username']}")

    print("\n📄 Inserting document...")
    user_id = user_data["id"]
    folder_id = "default-folder-123"

    insert_result = insert_document(
        text="This paper discusses neural networks in AI.",
        file_name="ai_paper.pdf",
        file_format="pdf",
        summary="An overview of neural networks",
        category="AI",
        user_id=str(user_id),  # Ensure it's a string (VARCHAR in schema)
        folder_id=folder_id
    )
    print(insert_result.get("message", "No message returned."))

    print("\n🔍 Searching for the document by filename...")
    results = search_by_filename("ai_paper.pdf")
    if results:
        for doc in results:
            print(f"📁 Found: {doc}")
    else:
        print("❌ No document found.")

    print("\n🗑️ Deleting the document...")
    delete_result = delete_document("ai_paper.pdf")
    print(delete_result.get("message", "No message returned."))

    print("\n🔍 Verifying deletion...")
    results = search_by_filename("ai_paper.pdf")
    if not results:
        print("✅ Document successfully deleted.")
    else:
        print("❌ Document still exists!")
