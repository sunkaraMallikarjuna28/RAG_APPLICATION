# test_all_packages.py
def test_all_installations():
    print("🔍 Testing All RAG Packages...")
    
    packages = [
        ("python-dotenv", "dotenv"),
        ("langchain", "langchain"),
        ("langchain-community", "langchain_community"),
        ("langchain-experimental", "langchain_experimental"),
        ("google-generativeai", "google.generativeai"),
        ("pymongo", "pymongo"),
        ("pypdf", "pypdf"),
        ("python-docx", "docx"),
        ("sentence-transformers", "sentence_transformers"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn")
    ]
    
    success_count = 0
    
    for package_name, import_name in packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}: Installed")
            success_count += 1
        except ImportError:
            print(f"❌ {package_name}: Failed")
    
    print(f"\n🎉 {success_count}/{len(packages)} packages installed successfully!")
    
    if success_count == len(packages):
        print("✅ ALL PACKAGES READY! You can proceed to next step.")
    else:
        print("❌ Some packages failed. Please check errors above.")

if __name__ == "__main__":
    test_all_installations()
