# testing.py

from collection_Control import (
    register_user,
    login_user,
    insert_document,
    search_by_filename,
    delete_document
)

print("\n🔐 Registering user...")
register_user("Sarah", "sarah@example.com", "securepass123")

print("\n🔐 Trying to register with duplicate email...")
register_user("Another", "sarah@example.com", "newpass")

print("\n🔐 Logging in with wrong password...")
login_user("sarah@example.com", "wrongpass")

print("\n🔐 Logging in with correct password...")
user_data = login_user("sarah@example.com", "securepass123")

if user_data:
    print("\n📄 Inserting document...")
    insert_document(
        text="This paper discusses neural networks in AI.",
        file_name="ai_paper.pdf",
        file_format="pdf",
        summary="An overview of neural networks",
        category="AI"
    )

    print("\n🔍 Searching for the document by filename...")
    results = search_by_filename("ai_paper.pdf")
    if results:
        for doc in results:
            print(f"📁 Found: {doc}")
    else:
        print("❌ No document found.")

    print("\n🗑️ Deleting the document...")
    delete_document("ai_paper.pdf")

    print("\n🔍 Verifying deletion...")
    results = search_by_filename("ai_paper.pdf")
    if not results:
        print("✅ Document successfully deleted.")
    else:
        print("❌ Document still exists!")
